"""VoxCom business domain: canonical enums, pure rules, spoken copy, the injected clock, and services.

``rules.py`` holds pure business rules and imports no provider, storage or
framework code (NFR-07). Services in ``services/`` reach data only through the
repository interfaces of :mod:`voxcare.storage`.
"""
