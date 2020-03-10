# Applied NLP reproduction project
Sentiment analysis of movie reviews 🎥🍿

## Reproduced paper
Pang, B. and Lee, L., 2005, June. *Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.* In Proceedings of the 43rd annual meeting on association for computational linguistics (pp. 115-124). Association for Computational Linguistics.

This paper can be found [here](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf).

## Datasets
This data was first used in Bo Pang and Lillian Lee,
*Seeing stars: Exploiting class relationships for sentiment categorization
with respect to rating scales.*, Proceedings of the ACL, 2005.

The datasets can be found [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).


## Running Locally

This section explains how to set up and run the code locally.

### Prerequisites

In our tests we used Python 3 (`3.7.6`) which is what we recommend you t
 install.
It is optional to create a virtualenv with this python version as described
 in step 2, this requires you to have virtualenv (`16.6.2`) installed on your
 system too.

### Installation steps

1. Clone the repository to a place of your liking.
2. `Optional` Create a virtual environment using
[virtualenv](https://virtualenv.pypa.io/en/stable/):
    ```bash
    # Create the virtualenv.
    python3 -m virtualenv --python=/location/to/python3.7.6/local/bin/python .venv

    # Activate the virtualenv.
    source .venv/bin/activate
    ```
3. Install the required dependencies specified in `requirements.txt`:
    ```bash
    python -m pip install -r requirements.txt
    ```

### Running steps

Given you have completed the installation steps and if you chose to create a
 virtualenv you have activated it you can run the following commands to run
 the prediction:

1. Run the models using:
    ```bash
    python predict.py
    ```

This command will start going through all the experiments described in our
 report.
