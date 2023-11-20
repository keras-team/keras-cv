sudo pip install --upgrade pip
sudo pip install -r requirements.txt --progress-bar off
sudo pip install -e ".[tests]"
sudo apt update
sudo apt install -y clang-format
