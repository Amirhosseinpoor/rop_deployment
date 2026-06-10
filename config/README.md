# Config: Global System Orchestration

The `config` module is the architectural backbone of the Mediverse AI platform, managing global settings, security middleware, and the integration of the hybrid Django/FastAPI ecosystem.

---

### 1. Settings & Governance (`settings.py`)
The configuration uses `django-environ` for strict 12-factor app compliance.

- **Authentication Pipeline**:
    - Integrates `allauth` with Google OAuth2 providers.
    - Configures `AUTH_PASSWORD_VALIDATORS` for enterprise-grade security.
- **Async Orchestration**:
    - Defines `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` (Redis) for managing long-running AI inference tasks in `single_rop` and `test_analysis`.
- **Database Architecture**:
    - Implements a PostgreSQL connection pool.
    - Configures `DATABASES` to utilize environment variables for sensitive credentials (`DB_NAME`, `DB_USER`, `DB_PASSWORD`).
- **Security & CORS**:
    - Defines `CSRF_TRUSTED_ORIGINS` for `arasai.ir` and subdomains.
    - Middleware stack includes `WhiteNoiseMiddleware` for optimized static asset serving and `AccountMiddleware` for session-aware authentication.

---

### 2. Global Routing & Entry Points (`urls.py`)
Acts as the top-level dispatcher for all modular applications.

| Prefix | Module | Responsibility |
| :--- | :--- | :--- |
| `admin/` | `jazzmin` | Custom-branded administrative dashboard. |
| `accounts/` | `allauth` | Social and local authentication flows. |
| `rop/` | `single_rop` | ROP diagnostic pipeline. |
| `double/` | `double_rop` | KC diagnostic pipeline. |
| `analysis/` | `test_analysis` | Agentic health auditing. |
| `market/` | `doctors_marketplace` | Clinician RAG & Studio. |
| `usac/` | `usac` | Identity & Company management. |

---

### 3. WSGI & ASGI Entry Points (`wsgi.py`, `asgi.py`)
- **`wsgi.py`**: Production entry point for Gunicorn, serving the synchronous Django components (ORM, Admin, Views).
- **`asgi.py`**: Asynchronous entry point, enabling future support for WebSockets (e.g., real-time AI inference status updates).

---

### 4. Middleware & Context Processors
- **`whitenoise`**: Handles static file compression and caching headers.
- **Custom Context Processors**: Injects global state (like user roles or company info) into the Django template engine for role-based UI rendering.

---
*Note: This module manages global orchestration. No local templates or static assets are owned by this module.*
