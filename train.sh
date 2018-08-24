#Train on flyingThings3D Dataset
python main.py --batch_size 8 --model DispNetF --optimizer=Adam --optimizer_lr=1e-4 --loss=MultiScale --loss_norm=L1 --validation_n_batches=10 \
--loss_numScales=7 --loss_startScale=1 --optimizer_lr=1e-4 --crop_size 384 768 \
--training_dataset FlyingThings --training_dataset_list_name /home/ruichao/Desktop/cnn-stereo/data/FlyingThings3D_release_TEST_screen.list  \
--validation_dataset FlyingThings --validation_dataset_list_name /home/ruichao/Desktop/cnn-stereo/data/FlyingThings3D_release_TEST_screen.list
