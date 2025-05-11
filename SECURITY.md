# Security Policy

## Supported Versions

We actively provide security updates for the following versions:

| Version | Supported |
| ------- | --------- |
| 5.1.x   | ✅ Yes     |
| 5.0.x   | ❌ No      |
| 4.0.x   | ✅ Yes     |
| < 4.0   | ❌ No      |

## Reporting a Vulnerability

To report a security vulnerability, especially if it involves sensitive files such as `firebase config.json`, please follow the responsible disclosure guidelines below:

* **Email us at:** `security@yourdomain.com` (replace with your actual contact)
* **Please include:**

  * A detailed description of the vulnerability
  * Steps to reproduce
  * Potential impact
  * Screenshots or logs if applicable

We aim to acknowledge your report within **48 hours**, provide progress updates every **5–7 business days**, and notify you of resolution status.

If a submitted vulnerability is verified:

* It will be patched promptly (typically within 14 days).
* Credit will be given unless anonymity is requested.

### 🔐 Firebase `config.json` Notice

If you discover misuse, accidental exposure, or abuse involving Firebase credentials (`config.json`), **please do not share it publicly**. Contact us immediately through the above reporting channel. Even though Firebase config files are safe to share *in general*, misuse can still occur if rules are not properly secured.
