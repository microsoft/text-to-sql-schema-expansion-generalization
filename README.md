# Bridging the Generalization Gap in Text-to-SQL Parsing with Schema Expansion 

Code and data release for this [ACL 2022 paper](https://aclanthology.org/2022.acl-long.381/).

## Schema Pruning 

Install pytorch 1.8.2 that fits your CUDA version. 

For schema pruning, the Synthetic dataset is under the folder data/syn_exp, and Squall data is data/syn_exp/train_data_prune.json and data/syn_exp/dev_data_prune.json

Training Command (for Squall): 

`python -m col-prune.experiment --train-file squall/train_data_prune.json --dev-file squall/dev_data_prune.json --test-file squall/dev_data_prune.json --log-file log_prune.log --pred-file pred_dev.json` 

Eval Command:

`python -m col-prune.experiment --train-file squall/train_data_prune.json --dev-file squall/dev_data_prune.json --test-file squall/dev_data_prune.json --log-file log_prune.log --pred-file pred_dev.json --test`

## Parsing (Seq2seq)

We follow the repo from the original Squall paper, see [here](https://github.com/tzshi/squall) for installation details.

The data used in parsing experiment is preprocessed where each table only contains remained columns after pruning. 
The Synthetic dataset is under folder data/syn_exp/after_prune, and Squall data is squall/train_data_parser.json and squall/dev_data_parser.json

Command:

Run `python main.py --train-file squall/train_data_parser.json --dev-file squall/dev_data_parser.json --test-file squall/dev_data_prune.json --log-file log_parse.log --pred-file pred_dev.json` 

## Parsing (SmBop)

We follow the repo from the original SmBop paper, see [here](https://github.com/OhadRubin/SmBop) for installation details.

The dataset in under folder smbop/dataset, and all config files are under folder configs/

Command:

`python exec.py --name squall --config_f configs/exp4.jsonnet`

# Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Note that our data release is effectively a repartitioning of the Squall dataset. The Squall dataset
is available [here](https://github.com/tzshi/squall) under a CC BY-SA 4.0 license, which is available
[here](https://github.com/tzshi/squall/blob/main/LICENSE-data). The Squall dataset was built upon
WikiTableQuestions by Panupong Pasupat and Percy Liang, available [here](https://github.com/ppasupat/WikiTableQuestions).
The WikiTableQuestions dataset is made available pursuant to CC BY-SA 4.0 as well.
See https://github.com/ppasupat/WikiTableQuestions/blob/master/LICENSE.

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
