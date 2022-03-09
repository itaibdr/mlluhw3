from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from transformers import RobertaForSequenceClassification



def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    prf = precision_recall_fscore_support(labels, preds, average = 'binary')
    acc = accuracy_score(labels, preds)
    
    scores = {
                "accuracy": acc,
                "f1": prf[2],
                "precision": prf[0],
                "recall": prf[1],
            }
        
    return scores

def model_init(params):
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    #if params is not None:
      #db_config.update({'dropout': params['dropout']})

    return RobertaForSequenceClassification.from_pretrained("roberta-base", return_dict = True)

    
    
