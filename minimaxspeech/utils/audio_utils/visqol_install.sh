# Reference: https://github.com/descriptinc/audiotools/issues/93#issuecomment-1650435399
apt update && apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

apt update && apt-get install -y --no-install-recommends bazel bazel-5.3.2

cd ~ && git clone https://github.com/google/visqol.git
cd visqol && bazel build :visqol -c opt
pip install .

# to address version `GLIBCXX_3.4.30' not found issue
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /opt/conda/lib/libstdc++.so.6