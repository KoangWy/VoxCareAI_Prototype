"""Voice rendering rewrites and spoken order-code normalisation."""

from __future__ import annotations

import pytest

from voxcare.session.speech import first_sentence, speakable
from voxcare.session.tools import normalize_spoken_order_codes


def test_speakable_rewrites_codes_dates_and_currency():
    spoken = speakable(
        "Order ORD-1002 is in transit with GHN. Estimated delivery date: 2026-10-15. Total 18,990,000 VND."
    )
    assert "O R D one zero zero two" in spoken and "October 15, 2026" in spoken and "18,990,000 dong" in spoken
    assert speakable("10,000 VND/km") == "10,000 dong per km"
    assert speakable("Clause (POL-WAR-001) applies") == "Clause applies"
    assert "T I C K, C S, eight eight two one" in speakable("ticket TICK-CS-8821")
    assert speakable("VOXCOM_500K") == "VOXCOM 500K"


def test_first_sentence():
    assert first_sentence("One. Two.") == "One."


@pytest.mark.parametrize(
    ("heard", "code"),
    [
        ("where is o r d one zero zero two", "ORD-1002"),
        ("check or rd one zero zero three", "ORD-1003"),
        ("order number ten oh two", "ORD-1002"),
        ("ord forty five sixty seven", "ORD-4567"),
    ],
)
def test_spoken_order_codes(heard, code):
    text, codes = normalize_spoken_order_codes(heard)
    assert codes == (code,) and code in text


def test_non_codes_are_left_alone():
    assert normalize_spoken_order_codes("order two TVs") == ("order two TVs", ())
    assert normalize_spoken_order_codes("order one two three four five")[1] == ()


def test_misheard_voxcom_terms_are_corrected():
    from voxcare.session.tools import normalize_spoken_terms

    heard = "compare the samson and the l g fifty five inch tv and check if voucher vox com five hundred k applies"
    text, notes = normalize_spoken_terms(heard)
    assert text == "compare the Samsung and the LG 55-inch tv and check if voucher VOXCOM_500K applies"
    assert len(notes) == 4
    assert normalize_spoken_terms("does box com tet five hundred k still work")[0] == "does VOXCOM_TET_500K still work"
    assert normalize_spoken_terms("my washer displays error e four")[0] == "my washer displays error E4"
    # ordinary words stay
    assert normalize_spoken_terms("he left for work and the box came") == ("he left for work and the box came", ())
