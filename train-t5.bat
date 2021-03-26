python seq2seq.py --model_name_or_path t5^
 --do_train --do_eval --task translation_transcribed_to_corrected --source_lang transcribed --target_lang corrected^
 --train_file dataset.json --validation_file dataset.json --output_dir asrcorrect^
 --per_device_train_batch_size=4 --per_device_eval_batch_size=4^
 --overwrite_output_dir --predict_with_generate^
 --num_train_epochs=50 --source_prefix "correct: "