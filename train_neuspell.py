from neuspell import BertChecker
checker = BertChecker()

checker.finetune(clean_file="neuspell-correct-dataset.txt", corrupt_file="neuspell-wrong-dataset.txt")
