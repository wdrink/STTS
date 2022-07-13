bash tools/dist_test.sh configs/Kinetics/t0_0.375.py ./checkpoints/t0_0.375.pth 8 --eval top_k_accuracy

bash tools/dist_test.sh configs/Kinetics/t0_0.5625.py ./checkpoints/t0_0.5625.pth 8 --eval top_k_accuracy

bash tools/dist_test.sh configs/Kinetics/t0_0.625.py ./checkpoints/t0_0.625.pth 8 --eval top_k_accuracy

bash tools/dist_test.sh configs/Kinetics/t0_0.75.py ./checkpoints/t0_0.75.pth 8 --eval top_k_accuracy

bash tools/dist_test.sh configs/Kinetics/t0_0.875.py ./checkpoints/t0_0.875.pth 8 --eval top_k_accuracy