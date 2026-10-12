#!/usr/bin/env bash
# One-time setup on a fresh Amazon Linux 2023 EC2 instance.
# Usage: bash ec2_bootstrap.sh https://github.com/<you>/rag-research-assistant.git
set -euo pipefail
REPO_URL="${1:?Pass your GitHub repo URL}"

# swap so a 1 GB instance survives the image build
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

git clone "$REPO_URL" ~/rag-research-assistant
cd ~/rag-research-assistant
cp .env.example .env && sed -i 's/^HOST_PORT=.*/HOST_PORT=80/' .env
echo "Done. Log out and back in, then: cd ~/rag-research-assistant && docker compose up -d --build"
