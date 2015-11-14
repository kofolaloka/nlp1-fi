# nlp1-fi
Project on Frame Induction for the 2015 course Natural Language Processing

## install missing python packages

run the following script as either `root` or normal user to install the packages with `pip`:
  ```
   bash install_packages.sh
  ```
## download dataset

run this script:
  ```
   python2 download_dataset.py destination_dir
  ```
  
## pre-processing of the downloaded data (will create a `dataset/` directory)
  ```
   python2 build_data.py --data-folder raw/ --output-folder .
  ```

## counting the triples: collapsing several partial counts contained in the pre-processed data
  ```
  python2 count.py -d dataset/ -c counted/
  ```
