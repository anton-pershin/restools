# Papers

Each paper we publish or plan to publish has the associated directory in `/papers` where codes for data abstractions, processing and plotting are stored. Here is a general structure of paper data:
```
papers
  |
  ------ paper_name_1
  |             |
  |             ----------- launchers
  |             |               |
  |             |               ----------- launcher_1.py
  |             |               |
  |             |               ----------- launcher_2.py
  |             |               |
  |             |               ----------- launcher_3.py
  |             |
  |             ----------- programs
  |             |               |
  |             |               ----------- program_1.py
  |             |               |
  |             |               ----------- program_2.py
  |             |               |
  |             |               ----------- program_3.py
  |             |
  |             ----------- views
  |             |             |
  |             |             ----------- figure_1.py
  |             |             |
  |             |             ----------- figure_2.py
  |             |             |
  |             |             ----------- figure_3.py
  |             |
  |             ----------- data.py
  |             |
  |             ----------- extensions.py
  |
  ------ paper_name_2
  |             |
  |             ----------- launchers
  |             |
  |             ----------- programs
  |             |
  |             ----------- views
  |             |
  |             ----------- data.py
  |             |
  |             ----------- extensions.py
  |
```
Let's describe this structure:
* `.../launchers`: each `.py` file in this directory can be used to launch a particular task either on a local or remote machine
* `.../programs`: each file in this directory is an executable (or a self-contained `.py` script) which can be used as a part of a task/algorithm/graph
* `.../views`: each file in this directory can be used to produce a figure (some of these figures can be found in the paper, some of them are exploratory)
* `.../data.py`: summary data representation implemented as `class Summary`; instances of this class are json-serializable and stored in `/jsons` 
* `.../extensions.py`: any other codes necessary to process research data (codes from this directory can migrate to `/restools`)
