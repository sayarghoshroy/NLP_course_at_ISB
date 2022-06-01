python run_experiment.py name tf_para_reddit_no_bal path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 0 bal 0
python run_experiment.py name tf_para_reddit_bal_1 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 0 bal 1
python run_experiment.py name tf_para_reddit_bal_2 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 0 bal 2

python run_experiment.py name tf_para_reddit_no_bal path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 1 bal 0
python run_experiment.py name tf_para_reddit_bal_1 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 1 bal 1
python run_experiment.py name tf_para_reddit_bal_2 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 1 bal 2

python run_experiment.py name reddit_no_bal path ../reddit test_mode 0 use_aug 0 model 0 bal 0
python run_experiment.py name reddit_bal_1 path ../reddit test_mode 0 use_aug 0 model 0 bal 1
python run_experiment.py name reddit_bal_2 path ../reddit test_mode 0 use_aug 0 model 0 bal 2

python run_experiment.py name reddit_no_bal path ../reddit test_mode 0 use_aug 0 model 1 bal 0
python run_experiment.py name reddit_bal_1 path ../reddit test_mode 0 use_aug 0 model 1 bal 1
python run_experiment.py name reddit_bal_2 path ../reddit test_mode 0 use_aug 0 model 1 bal 2
