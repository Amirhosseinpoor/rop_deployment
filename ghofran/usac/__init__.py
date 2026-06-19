"""USAC — User, Company & Access-Control microservice.

This is the identity backbone of the SurgiNote ecosystem, ported from the Django
``usac`` app. It owns:

* **Users & authentication** — registration and JWT-based login.
* **Companies** — each manager owns exactly one company.
* **Role-based access** — manager / doctor / employee.
* **Invitations** — managers invite staff by national code; staff may only sign
  up if a matching, unused invitation exists.

The original Django app also rendered manager dashboards and exported CSV history
by reading tables that belong to *other* services (single_rop / double_rop /
test_analysis). To keep this microservice independent, those cross-service
analytics are intentionally out of scope here — see ``HOW_TO_RUN.md`` for how the
ecosystem is expected to compose them.
"""
