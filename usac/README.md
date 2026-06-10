# USAC: User Systems & Account Control

The `usac` module provides the foundational identity and governance layer for the Mediverse AI platform, managing multi-tenant company structures and role-based access.

---

### 1. Models (`models.py`)
- **`Company`**:
    - Represents a medical center or industrial client.
    - **`manager` (OneToOne)**: Links a company to a primary administrative user.
- **`UserProfile`**:
    - Extends Django's `auth.User` with a `role` (Manager, Doctor, Employee) and `national_code`.
    - **`company` (FK)**: Groups users into their respective organizational tenants.
- **`Invitation`**:
    - A pre-registration model used by managers to white-list employees and doctors based on their `national_code`.

---

### 2. Core Logic & Signals (`signals.py`)
The module uses Django signals to maintain data integrity across the platform.

- **`create_profile_for_user`**: Automatically initializes a `UserProfile` whenever a new Django user is created (e.g., via Social Auth).
- **`sync_group_membership`**: Synchronizes Django's internal `auth.Group` system with the `UserProfile.role`, enabling easy permission management in the Django Admin.

---

### 3. Views & Handlers (`views.py`)
- **`role_based_redirect`**: A core middleware-like handler that routes users to their specific dashboard (`manager_dashboard`, `doctor_dashboard`, or the `dilemma` page) upon login.
- **`manager_dashboard`**:
    - **Tenant Management**: Allows managers to invite users and see their registration status.
    - **Analytics Engine**: Computes real-time health prevalence statistics (Diabetes, Smoking, BMI distribution) for the company's workforce using Django ORM aggregation.
- **`export_history_csv`**:
    - A technical exporter that generates a unified CSV of all ROP and KC diagnostics performed by the user.
- **Admin Exporters**:
    - `export_misclassified_rop_csv` and `export_misclassified_kc_csv` provide the AI engineering team with raw datasets of failed inferences for model retraining.

---

### 4. Routing (`urls.py`)
| Endpoint | View | Logic |
| :--- | :--- | :--- |
| `/` | `custom_login` | Primary authentication entry. |
| `/dilemma/` | `dilemma_view` | Role-selection or primary landing for employees. |
| `/managing/` | `manager_dashboard` | Multi-tenant admin panel & analytics. |
| `/history/export/` | `export_history_csv` | Diagnostic record download. |
| `/export/rop/` | `export_misclassified_rop_csv` | Admin data dump for ROP. |

---
*Note: This module handles its corresponding frontend views via template rendering in the `templates/usac/` directory.*
