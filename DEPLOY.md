# Deploy to AWS EC2

**Cost:** accounts created before 15 July 2025 get 12 months of a free `t2/t3.micro`; newer accounts get credits (up to about $200, Free Plan up to 6 months). Check https://aws.amazon.com/free and set a billing budget alert first.

**Ollama:** a micro instance (1 GB RAM) can't run an Ollama model, so the public demo runs in extractive mode (best passages with sources). For LLM answers, run locally or use a larger paid instance with Ollama installed (set `OLLAMA_MODEL` in `.env`, and set `OLLAMA_HOST=0.0.0.0` on the Ollama service). Stop paid instances after demos.

## Steps

1. **Launch** an EC2 instance: Amazon Linux 2023, `t3.micro`, a new key pair (never commit the `.pem`), 8–16 GB storage. Security group: SSH (22) from *My IP*, HTTP (80) from anywhere.
2. **Bootstrap** (one time):
   ```bash
   ssh -i key.pem ec2-user@<public-ip>
   curl -O https://raw.githubusercontent.com/Kartikey2413/rag-research-assistant/main/deploy/ec2_bootstrap.sh
   bash ec2_bootstrap.sh https://github.com/Kartikey2413/rag-research-assistant.git
   exit        # log back in so the docker group applies
   ```
3. **Run:**
   ```bash
   ssh -i key.pem ec2-user@<public-ip>
   cd ~/rag-research-assistant && docker compose up -d --build
   ```
   Open `http://<public-ip>/`. Check `curl http://<public-ip>/health`.
4. **Update** after pushing changes: `git pull && docker compose up -d --build`.

## Notes

- The public IP changes when the instance is stopped and started; an Elastic IP keeps it fixed (public IPv4 is billed hourly).
- Traffic is plain HTTP. Put Caddy or a load balancer with HTTPS in front for anything sensitive.
- The same `Dockerfile` runs on Render or Fly.io if credits run out.
