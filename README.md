# nlp1-fi
Project on Frame Induction for the 2015 course Natural Language Processing

## install missing python packages

run the following script as either `root` or normal user to install the packages with `pip`:
  ```
   bash install_packages.sh
```

## missing `nltk` corpora

during running the scripts there will be probably a few exceptions launched that will tell 
you which additional corpora need to be retrieved.

Every time that this happens, you have to fix it by launching the python console:

  ```
   python2
  ```

and typing the following:

  ```
   import nltk
   nltk.download()
  ```

This will launch the GUI for downloading the corpora.

## download dataset

run this script:
  ```
   python2 download_dataset.py destination_dir
  ```
  
## pre-processing of the downloaded data 
  This will create a `dataset/` directory and convert all verb forms to the infinitive by using WordNet's lemmatizer
  ```
   python2 build_data.py --data-folder raw/ --output-folder .
  ```

## counting the triples
collapse several partial counts with following command:
  ```
   python2 count.py -i ./dataset/ -o ./counted/
  ```
  
## sorting the triples by count
  ```
   python sort.py -i ./counted/ -o ./sorted/
  ```

## most frequent across the entire dataset
  ```
  python most_frequent.py -i ./sorted/ -o ./most_frequent/ -n 4000
  ```
