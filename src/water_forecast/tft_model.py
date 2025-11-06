from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

def make_tft(training_dataset,
             hidden_size=160,
             attention_heads=4,
             dropout=0.1,
             learning_rate=1e-3,
             quantiles=(0.1, 0.5, 0.9)):

    loss = QuantileLoss(quantiles=quantiles)

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        loss=loss,
        output_size=len(quantiles),
        log_interval=50,
        reduce_on_plateau_patience=4
    )

    # Ignore các nn.Module attributes khi checkpointing
    model.save_hyperparameters(ignore=["loss", "logging_metrics"])

    return model

