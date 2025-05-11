from transformers import AutoConfig

config = AutoConfig.from_pretrained("HURIDOCS/pdf-document-layout-analysis")
print(config)
