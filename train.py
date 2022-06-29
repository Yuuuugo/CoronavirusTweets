import sklearn
from dataprocessing import load_data
from function import conf_matrix
from models.multinomial import MultinomialNB_model
from models.BERT import create_model
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    X_train, X_valid, y_train, y_valid = load_data(transformers=False)
    model = MultinomialNB_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    print(f"Accuracy score : {accuracy_score(y_valid, y_pred)}")

    (
        train_input_ids,
        val_input_ids,
        train_attention_masks,
        val_attention_masks,
        y_train,
        y_valid,
    ) = load_data(transformers=True)
    model = create_model()

    history_bert = model.fit(
        [train_input_ids, train_attention_masks],
        y_train,
        validation_data=([val_input_ids, val_attention_masks], y_valid),
        epochs=4,
        batch_size=32,
    )
