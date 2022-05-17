# English to Hindi Transliteration

## Approach

We are using Seq-Seq model with attention on character level for transliteraion instead of word level that we use in translation. Architecture basically consist of encoder and decoder LSTM's.

## USAGE

- clone the repository
- create virtual environment and activate it using following command

```
cd Transliteration
python -m venv env
.\env\Scripts\activate
```

- install required dependencies

```
pip install -r requirements.txt
```

## Training

Run the following command for training (if you dont want to use pretrained model that I've already updated)

```
python train.py
```

## Inference

There are two methods

#### 1. Flask Based REST API on localhost

- run following command

```
python app.py
```

- for prediction use this curl request

```
curl --location --request POST 'http://127.0.0.1:5000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
   "message":"hello"
}'
```

#### 2. Command line Method

- Run this from your terminal

```
python .\inference.py --message "hello"
```

## Evaluation

- Mean F Score on Train Data - 0.9714
- Mean F Score on Validation Data - 0.5765

for more info check output/train_prediction.xlsx & output/validation_prediction.xlsx files

## Results

```
{
   "original_text": "hello",
   "transliterated_text": "हेल्लो"
}
```

```
{
  "original_text": "dawson",
  "transliterated_text": "डॉसन"
}
```

```
{
  "original_text": "darbhanga junction",
  "transliterated_text": "दरभंगा जंक्शन"
}
```

```
{
  "original_text": "chill",
  "transliterated_text": "चिल"
}
```

```
{
   "original_text": "tata",
   "transliterated_text": "टाटा"
}
```

## Improvements

- Instead of LSTM's we can try transformer which has better learning capabilities.

## Credits

- https://www.youtube.com/watch?v=EoGUlvhRYpk
- https://github.com/skshashankkumar41/Language-Translation-Using-PyTorch
