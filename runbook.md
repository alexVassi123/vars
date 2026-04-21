# Runbook

Korte documentatie van hoe ik alle trainingsruns heb uitgevoerd. Drie methoden (mean, attention, cross_attention) met elk drie seeds (1, 2, 3), in totaal 9 runs van ongeveer 6 uur op een RTX 4090.

## Smoke test (1 epoch)

Voor elke methode eerst een run van 1 epoch om te checken dat er niks crasht, de loss daalt en er een `best_model.pth.tar` wordt opgeslagen.

```bash
python "VARS model/main.py" \
  --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls \
  --num_views 2 --pooling_type <METHODE> --pre_model r2plus1d_18 \
  --batch_size 8 --max_epochs 1 --max_num_worker 16 \
  --LR 1e-4 --weight_decay 0.01 --patience 12 --GPU 0 \
  --start_frame 20 --end_frame 100 --fps 10 \
  --model_name SMOKE_<METHODE> --seed 1
```

Vervang `<METHODE>` door `mean`, `attention` of `cross_attention`.

## Volledige trainingsruns

Voor elke methode drie runs met seeds 1, 2 en 3. Hetzelfde commando, alleen `--pooling_type`, `--seed` en `--model_name` wisselen.

```bash
python "VARS model/main.py" \
  --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls \
  --num_views 2 --pooling_type <METHODE> --pre_model r2plus1d_18 \
  --batch_size 8 --max_epochs 40 --max_num_worker 16 \
  --LR 1e-4 --weight_decay 0.01 --patience 12 --GPU 0 \
  --start_frame 20 --end_frame 100 --fps 10 \
  --model_name THESIS_<METHODE>_s<SEED> --seed <SEED> \
  2>&1 | tee thesis_<METHODE>_s<SEED>.log
```



## Per run controleren

- Trainingslog toont dalende loss en stijgende validation LB
- `best_model.pth.tar` opgeslagen op een epoch tussen 10 en 25
- Beste val LB is geen NaN en blijft niet hangen op een laag plateau

## Uitvoer

Elke run schrijft naar `models/<model_name>/2/r2plus1d_18/0.0001/_B8_F32_S_G0.1_Step3/`:

- `best_model.pth.tar` met het gewicht op de beste validation epoch
- Per-epoch prediction JSONs voor validation en test


Logs gaan naar `logs/` en samengevatte resultaten staan in `results.csv`.
