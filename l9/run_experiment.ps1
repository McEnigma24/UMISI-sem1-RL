$env:HF_SSL_VERIFY = '0'

python train.py --cpu --n-train 500 --max-steps 150 --log-every 25 --no-anti-hack --name baseline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python train.py --cpu --n-train 500 --max-steps 150 --log-every 25 --name antihack
exit $LASTEXITCODE
