# How to Run USAC Service

**USAC** = Users, Companies & Access-Control. It is the identity backbone of the
SurgiNote ecosystem: registration, JWT login, role management, and the
manager-issued invitation flow that gates staff signup.

## Prerequisites
- Python 3.10+
- A database. By default it uses a local **SQLite** file (`usac.db`) created
  automatically — zero setup. For production point `USAC_DATABASE_URL` at
  PostgreSQL.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration (environment variables)

| Variable                   | Default                 | Description                                        |
|----------------------------|-------------------------|----------------------------------------------------|
| `USAC_DATABASE_URL`        | `sqlite:///./usac.db`   | SQLAlchemy URL. e.g. `postgresql+psycopg2://u:p@h/db` |
| `USAC_JWT_SECRET`          | *(insecure dev default)*| **Set this in production.** JWT signing secret.    |
| `USAC_JWT_ALGORITHM`       | `HS256`                 | JWT algorithm.                                     |
| `USAC_ACCESS_TOKEN_MINUTES`| `720`                   | Access-token lifetime (minutes).                   |
| `USAC_PORT`                | `8003`                  | Port for the convenience launcher.                 |

```bash
export USAC_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

## Run
From inside the `ghofran/` directory (so the package import resolves):
```bash
cd ..                    # into ghofran/
uvicorn usac.routes:app --host 0.0.0.0 --port 8003
```
Or self-launch (honours `USAC_PORT`):
```bash
cd usac
python routes.py
```
Tables are created automatically on first start. Interactive docs: <http://localhost:8003/docs>

## Test

**1. Liveness**
```bash
curl http://localhost:8003/health
```

**2. Register a manager (also creates their company)**
```bash
curl -X POST http://localhost:8003/signup/manager -H 'Content-Type: application/json' -d '{
  "username": "boss", "email": "boss@acme.com", "password": "supersecret1",
  "phone": "0912...", "company_name": "Acme Clinic", "company_address": "1 Main St",
  "company_email": "info@acme.com", "company_phone": "021..."
}'
```

**3. Log in (get a JWT)**
```bash
TOKEN=$(curl -s -X POST http://localhost:8003/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"boss","password":"supersecret1"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
echo "$TOKEN"
```

**4. Manager invites an employee by national code**
```bash
curl -X POST http://localhost:8003/manager/invitations \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"national_code":"1234567890","role":"employee"}'
```

**5. The invited employee registers**
```bash
curl -X POST http://localhost:8003/signup/employee -H 'Content-Type: application/json' -d '{
  "username":"alice","email":"alice@acme.com","password":"anothersecret1",
  "national_code":"1234567890"
}'
```

**6. View the manager dashboard (membership + invitations)**
```bash
curl http://localhost:8003/manager/dashboard -H "Authorization: Bearer $TOKEN"
```

## Scope & integration notes
This service deliberately covers **only** identity/access concerns. The original
Django app also produced:

- **Prediction history & CSV exports** — these read `single_rop` / `double_rop`
  records, which now live in those services. Expose history from there, or build
  an aggregation endpoint at the API gateway.
- **Manager dashboard health analytics** (BMI / risk / opinion charts) — these
  read `test_analysis` health profiles; compose them from that service.

`/auth/login` returns a `redirect` hint (`manager_dashboard` / `doctor_dashboard`
/ `dilemma`) so the frontend can route by role exactly as the Django app did.
```
