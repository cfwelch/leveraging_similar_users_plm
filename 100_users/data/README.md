generate_txt_data.py: generates posts_txt ending with '<eos>' from posts_json for anchor users 
prepare_data_for_anchor_users.py: generates 250000_tokens_of_anchor_users from anchor_posts_txt
cat 250000_tokens_of_anchor_users/* > posts_of_anchor_users.txt
prepare_data_for_validation_users.py: generates 250000_tokens_of_validation_users from validation_posts_txt
