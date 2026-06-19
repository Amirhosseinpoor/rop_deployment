"""Double-ROP (binocular keratoconus / corneal classification) microservice.

Given a *pair* of eye images (left + right), the ``EyeNet`` model produces:

* a per-eye class (Normal / ATN / NEIr / EIr / eKCN), and
* a combined "Z" class (SfRS / NSfRS) derived from both eyes together.

As with the other services the package separates the network architecture
(:mod:`model`) and the framework-free inference logic (:mod:`service`) from the
thin FastAPI layer (:mod:`routes`).
"""
