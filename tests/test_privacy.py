import unittest

from voxcare.privacy import mask_pii


class MaskPiiTests(unittest.TestCase):
    def test_self_introduced_names_are_masked(self):
        for text in ("My name is John Smith", "I'm John Smith", "I am Robert James Lee", "this is Mary"):
            with self.subTest(text=text):
                self.assertIn("[REDACTED]", mask_pii(text))
                self.assertNotIn("John Smith", mask_pii(text))
        self.assertNotIn("Robert James Lee", mask_pii("I am Robert James Lee"))

    def test_card_variants_require_luhn(self):
        for text in ("4111 1111 1111 1111", "4111-1111-1111-1111", "4111.1111.1111.1111"):
            with self.subTest(text=text):
                self.assertEqual(mask_pii(f"card {text} ok"), "card [REDACTED_NUMBER] ok")
        self.assertIn("1234 5678 9012 3456", mask_pii("card 1234 5678 9012 3456 ok"))

    def test_dob_ssn_address_and_country_phones_are_masked(self):
        self.assertIn("[REDACTED_DOB]", mask_pii("born 03/15/1990"))
        self.assertIn("[REDACTED_SSN]", mask_pii("ssn 123-45-6789"))
        self.assertIn("[REDACTED_ADDRESS]", mask_pii("I live at 123 Nguyen Trai street"))
        self.assertIn("[REDACTED_ADDRESS]", mask_pii("deliver to 45 Le Loi boulevard"))
        self.assertIn("[REDACTED_PHONE]", mask_pii("call +84909012345"))
        self.assertIn("[REDACTED_PHONE]", mask_pii("call +1 (415) 555-0134"))

    def test_reference_codes_and_plain_numbers_survive(self):
        text = (
            "order ORD-1001 shipment SHIP-1002 clause POL-WAR-001 sku TV-SAM-55 "
            "voucher VOXCOM_500K tracking GHN-VN-882910 total 11490000 on 2026-10-15"
        )
        self.assertEqual(mask_pii(text), text)
        self.assertEqual(mask_pii("driver +8490***45"), "driver +8490***45")

    def test_adversarial_spacing_and_mixed_content(self):
        masked = mask_pii("Hi I'm John Smith, call 4111 1111 1111 1111 about ORD-1002")
        self.assertIn("[REDACTED]", masked)
        self.assertIn("[REDACTED_NUMBER]", masked)
        self.assertIn("ORD-1002", masked)
        self.assertNotIn("John Smith", masked)
        self.assertNotIn("4111 1111 1111 1111", masked)


if __name__ == "__main__":
    unittest.main()


def test_reference_codes_survive_label_masking():
    assert mask_pii("Hi, this is ORD-1002") == "Hi, this is ORD-1002"
    assert mask_pii("this is POL-RET-001 right?") == "this is POL-RET-001 right?"
    assert mask_pii("I am TV-SAM-55 shopping") == "I am TV-SAM-55 shopping"
    assert mask_pii("My name is Linh Tran about ORD-1002") == "My name is [REDACTED] about ORD-1002"
