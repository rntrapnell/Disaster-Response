Overview and motivation

Social media has made communication in today's world easier and broader while simultaneously making it more challenging and complex. In a disaster people in need of help will send out pleas via social media. This is an improvement over old methods in that some with a phone would need a specific number to call for help and without knowing who to call might not ever ask for help. But it presents a new issue: how does the cry for help get to the right disaster relief organization?

What we have attempted to build here is a pipeline that can be trained on message data and deployed to sort individual messages into categories.

Files

process_data.py An ETL pipeline that takes two CSVs files (messages and categories)
                messages and prepares it to be used for training the classifier.
train_classifier.py Takes a SQL db (usually one created by process_data.py) and uses it to train the classifier.
run.py A web app that allows the user to input a message and receive back a category predicted by the classifier             

Instructions
The ETL pipeline is run from process_data.py and requires an input csv name as well as the name of an output sql db.
Example:
`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
The classifier is trained using train_classifier.py and requires an input sql db name (which is typically the output name above for process_data.py) and an output pickle (.pkl) file name(the web app only supports the name classifier.pkl so if you plan to use the web app you should name the pickle file accordingly).
Example:
`python train_classifier.py DisasterResponse.db classifier.pkl`

The web app is run using run.py and requires no inputs
`python run.py`
Once running it can be accessed by visiting localhost:3001 in a web browser

Evaluation and discussion

The model works decently but not great. The biggest problem we have is the data which lacks enough positive examples in certain categories to be useful (it was not uncommon for the testing or validation sets to lack an positives in certain categories).

The web app itself while somewhat helpful is probably not the best deployment for the model. It works fine for showing off the model but in the real world it's not practical as it requires as much human effort as, if not more than, classifying the messages by hand.

If the oppotunity exists to revisit this project in the future it would be interesting to try the project with a different embedding. I'm particularly interested in trying a sentence embedding such as InferSent https://github.com/facebookresearch/InferSent.
