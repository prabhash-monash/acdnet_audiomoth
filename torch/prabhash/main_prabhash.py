# This is written to understand what ACDNet has been originally written.
import glob
import os

import torch.cuda

import common_prabhash.opts_pra as opts_pra




if __name__ == "__main__":
    opts = opts_pra.parse();
    opts_pra.display_info(opts);
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

    user_input_flg_scratch_train = input("Press [s] to train from scratch. Otherwise, will train from existing model. s:")
    opts.retrain = True if user_input_flg_scratch_train == 's' else False;
    print(f"opts.retrain = {opts.retrain}");


    flg_valid_path_model = False;

    while(not flg_valid_path_model):
        user_input_path = input("AI-model's relative path (blank for ACDNet) : \n");

        if(user_input_path == ""):
            opts.model_path = "ACDNet";
            flg_valid_path_model = True;
            print(f"DEFAULT LOADED: {opts.model_path}")
        else:
            if(not user_input_path.startswith("/")):
                full_path = os.path.join(os.getcwd(), user_input_path);
            else:
                full_path = user_input_path;

            if(len(full_path)>0 and os.path.isfile(full_path)):
                state = torch.load(full_path,map_location=opts.device);
                print(f"pytorch model exists at : {full_path}");
                flg_valid_path_model = True;
                print(f"state = {state}")

            else:
                print(f'ERROR: {full_path} path does not exist!')

