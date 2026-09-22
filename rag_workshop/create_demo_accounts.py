"""
Generate one personalized rag_workshop/.streamlit/secrets.toml per participant for
streamlit_app_secure.py's demo-mode login (Section 14.3).

Why per-participant files: each participant runs their OWN Streamlit instance on their OWN
laptop (see notebook 2's distributed architecture) -- nobody ever logs into anyone else's local
instance. So each participant's file only needs to contain THEIR OWN username/password/LiteLLM
key, never the whole group's -- unlike the illustrative alice/bob example in Section 14.3, which
puts everyone in one shared secrets.toml. This also combines with the LiteLLM key each
participant was already issued by manage_litellm.sh create-keys.

Run with:  python3 rag_workshop/create_demo_accounts.py
Reads:     participant-keys.tsv (repo root -- see manage_litellm.sh create-keys)
Writes:    rag_workshop/participant-secrets/<alias>.toml            (one per participant)
           rag_workshop/participant-secrets/_distribution-list.tsv  (plaintext passwords, for
                                                                      you to hand out one row at
                                                                      a time -- never the whole
                                                                      file)
Both outputs are gitignored. Passwords are never stored in clear anywhere except that one
distribution list, which you should delete once everyone has their credentials.
"""
import os
import secrets
import sys

import streamlit_authenticator as stauth

WORKDIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WORKDIR)

KEYS_TSV = os.path.join(REPO_ROOT, "participant-keys.tsv")
OUTPUT_DIR = os.path.join(WORKDIR, "participant-secrets")

# Synthetic -- a walk-in workshop's participant-XX aliases don't come with known real emails,
# and streamlit_app_secure.py's per-user LiteLLM key lookup (14.1) is keyed by email.
EMAIL_DOMAIN = "labobots.workshop"

# Unambiguous when read aloud or handwritten: no 0/O, 1/l/I, no symbols.
PASSWORD_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"
PASSWORD_LENGTH = 10


def generate_password() -> str:
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def display_name(alias: str) -> str:
    return alias.replace("-", " ").title()  # "participant-07" -> "Participant 07"


def load_participant_keys(path: str) -> list[tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run ./rag_workshop/manage_litellm.sh create-keys first, "
            "then scp participant-keys.tsv down (see the README)."
        )
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            alias, key = line.split("\t")
            rows.append((alias, key))
    return rows


def render_secrets_toml(alias: str, email: str, litellm_key: str, password_hash: str, cookie_key: str) -> str:
    return f'''# Auto-generated for {alias} by rag_workshop/create_demo_accounts.py.
# Personal file -- do NOT share, do NOT commit (already gitignored).
# Copy this to rag_workshop/.streamlit/secrets.toml, then:
#   streamlit run rag_workshop/streamlit_app_secure.py

chroma_host = "localhost"
chroma_port = 8000
chroma_collection_name = "ccin2p3_docs"

litellm_proxy_url = "http://localhost:4000"
litellm_model_name = "workshop-llm"
litellm_key = "{litellm_key}"

auth_mode = "demo"

[user_keys]
"{email}" = "{litellm_key}"

[demo_auth]
cookie_name = "labobots_demo_auth"
cookie_key = "{cookie_key}"
cookie_expiry_days = 1

[demo_auth.credentials.usernames.{alias}]
name = "{display_name(alias)}"
email = "{email}"
password = "{password_hash}"
'''


def main():
    rows = load_participant_keys(KEYS_TSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    distribution_rows = []
    for alias, litellm_key in rows:
        email = f"{alias}@{EMAIL_DOMAIN}"
        password = generate_password()
        password_hash = stauth.Hasher.hash(password)
        cookie_key = secrets.token_hex(16)

        toml_text = render_secrets_toml(alias, email, litellm_key, password_hash, cookie_key)
        out_path = os.path.join(OUTPUT_DIR, f"{alias}.toml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(toml_text)
        os.chmod(out_path, 0o600)

        distribution_rows.append((alias, password))
        print(f"Generated {out_path}")

    dist_path = os.path.join(OUTPUT_DIR, "_distribution-list.tsv")
    with open(dist_path, "w", encoding="utf-8") as f:
        f.write("alias\tstreamlit_password\n")
        for alias, password in distribution_rows:
            f.write(f"{alias}\t{password}\n")
    os.chmod(dist_path, 0o600)

    print(f"\nGenerated {len(rows)} participant secrets.toml files in {OUTPUT_DIR}/")
    print(f"Distribution list (plaintext passwords -- hand out one row at a time, then delete): {dist_path}")


if __name__ == "__main__":
    main()
