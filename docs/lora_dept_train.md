IT support traing params: 

def main():
	base_model = "Qwen/Qwen2-0.5B"
	dataset_path = "finance"
	text_field = "text"
	response_field = "response"
	response_separator = "\n### Response:\n"
	peft_output_dir = "qwen_dept_lora_instruction"
	num_train_epochs = 3
	per_device_train_batch_size = 2
	gradient_accumulation_steps = 8
	learning_rate = 5e-5
	lora_r = 16
	lora_alpha = 32
	max_length = 512
	gradient_checkpointing = True
	fp16 = True
	bf16 = False
	warmup_steps = 50
	weight_decay = 0.01
	instruction_residual_path = None
	instruction_residual_repo = "Gaurav2k/qwen2-0.5b-instruction-residuals"
	instruction_residual_filename = "qwen2-0.5b-instruction-residuals.pt"

	class Args:
		def __init__(self):
			self.base_model = base_model
			self.dataset = dataset_path
			self.text_field = text_field
			self.response_field = response_field
			self.prompt_template = "{text}"
			self.response_separator = response_separator
			self.peft_output_dir = peft_output_dir
			self.train_split = "train"
			self.eval_split = None
			self.max_length = max_length
			self.per_device_train_batch_size = per_device_train_batch_size
			self.per_device_eval_batch_size = 1
			self.gradient_accumulation_steps = gradient_accumulation_steps
			self.learning_rate = learning_rate
			self.weight_decay = weight_decay
			self.num_train_epochs = num_train_epochs
			self.warmup_steps = warmup_steps
			self.logging_steps = 10
			self.save_steps = 500
			self.eval_steps = 500
			self.evaluation_strategy = "no"
			self.lora_r = lora_r
			self.lora_alpha = lora_alpha
			self.lora_dropout = 0.05
			self.lora_bias = "none"
			self.lora_target_modules = None
			self.device_map = "auto"
			self.dtype = "auto"
			self.fp16 = fp16
			self.bf16 = bf16
			self.gradient_checkpointing = gradient_checkpointing
			self.seed = 42
			self.max_train_samples = None
			self.max_eval_samples = None
			self.num_proc = None
			self.dataset_config = None
			self.instruction_residual_path = str(instruction_residual_path) if instruction_residual_path else None
			self.instruction_residual_repo = instruction_residual_repo
			self.instruction_residual_filename = instruction_residual_filename
			self.hf_token = settings.HF_TOKEN
			self.preprocess = None

	config = Args()
	residual_state = load_instruction_residual(
		config.instruction_residual_path,
		config.instruction_residual_repo,
		config.instruction_residual_filename,
		config.hf_token,
	)

	selected_dataset = config.dataset.lower()
	if selected_dataset in DEPARTMENT_DATASETS:
		spec = DEPARTMENT_DATASETS[selected_dataset]
		dept_args = deepcopy(config)
		dept_args.dataset = spec["dataset"]
		dept_args.text_field = spec.get("text_field", dept_args.text_field)
		dept_args.response_field = spec.get("response_field", dept_args.response_field)
		dept_args.prompt_template = spec.get("prompt_template", dept_args.prompt_template)
		dept_args.response_separator = spec.get("response_separator", dept_args.response_separator)
		dept_args.train_split = spec.get("train_split", dept_args.train_split)
		dept_args.eval_split = spec.get("eval_split", dept_args.eval_split)
		dept_args.dataset_config = spec.get("dataset_config", dept_args.dataset_config)
		dept_args.preprocess = spec.get("preprocess")
		if "max_length" in spec:
			dept_args.max_length = spec["max_length"]
		if "gradient_accumulation_steps" in spec:
			dept_args.gradient_accumulation_steps = spec["gradient_accumulation_steps"]
		if "max_train_samples" in spec:
			dept_args.max_train_samples = spec["max_train_samples"]
		if "max_eval_samples" in spec:
			dept_args.max_eval_samples = spec["max_eval_samples"]
		dept_output_dir = spec.get("output_dir", f"{config.peft_output_dir}_{selected_dataset}")
		dept_args.peft_output_dir = dept_output_dir
		train_single_department(dept_args.dataset, dept_output_dir, dept_args, residual_state)
		return

	train_single_department(config.dataset, config.peft_output_dir, config, residual_state)



Finance 

def main():
	base_model = "Qwen/Qwen2-0.5B"
	dataset_path = "finance"
	text_field = "text"
	response_field = "response"
	response_separator = "\n### Response:\n"
	peft_output_dir = "qwen_dept_lora_instruction_finance"
	num_train_epochs = 2
	per_device_train_batch_size = 2
	gradient_accumulation_steps = 8
	learning_rate = 3e-5
	lora_r = 16
	lora_alpha = 32
	max_length = 512
	gradient_checkpointing = True
	fp16 = True
	bf16 = False
	warmup_steps = 50
	weight_decay = 0.001
	instruction_residual_path = None
	instruction_residual_repo = "Gaurav2k/qwen2-0.5b-instruction-residuals"
	instruction_residual_filename = "qwen2-0.5b-instruction-residuals.pt"

	class Args:
		def __init__(self):
			self.base_model = base_model
			self.dataset = dataset_path
			self.text_field = text_field
			self.response_field = response_field
			self.prompt_template = "{text}"
			self.response_separator = response_separator
			self.peft_output_dir = peft_output_dir
			self.train_split = "train"
			self.eval_split = None
			self.max_length = max_length
			self.per_device_train_batch_size = per_device_train_batch_size
			self.per_device_eval_batch_size = 1
			self.gradient_accumulation_steps = gradient_accumulation_steps
			self.learning_rate = learning_rate
			self.weight_decay = weight_decay
			self.num_train_epochs = num_train_epochs
			self.warmup_steps = warmup_steps
			self.logging_steps = 10
			self.save_steps = 500
			self.eval_steps = 100
			self.evaluation_strategy = "no"
			self.lora_r = lora_r
			self.lora_alpha = lora_alpha
			self.lora_dropout = 0.05
			self.lora_bias = "none"
			self.lora_target_modules = None
			self.device_map = "auto"
			self.dtype = "auto"
			self.fp16 = fp16
			self.bf16 = bf16
			self.gradient_checkpointing = gradient_checkpointing
			self.seed = 42
			self.max_train_samples = 3000
			self.max_eval_samples = None
			self.num_proc = None
			self.dataset_config = None
			self.instruction_residual_path = str(instruction_residual_path) if instruction_residual_path else None
			self.instruction_residual_repo = instruction_residual_repo
			self.instruction_residual_filename = instruction_residual_filename
			self.hf_token = settings.HF_TOKEN
			self.preprocess = DEPARTMENT_DATASETS["finance"].get("preprocess")


Engineering
base_model = "Qwen/Qwen2-0.5B"
	dataset_path = "engineering"
	text_field = "text"
	response_field = "response"
	response_separator = "\n### Response:\n"
	peft_output_dir = "qwen_dept_lora_instruction_engineering"
	num_train_epochs = 2
	per_device_train_batch_size = 2
	gradient_accumulation_steps = 8
	learning_rate = 3e-5
	lora_r = 16
	lora_alpha = 32
	max_length = 512
	gradient_checkpointing = True
	fp16 = True
	bf16 = False
	warmup_steps = 50
	weight_decay = 0.001
	instruction_residual_path = None
	instruction_residual_repo = "Gaurav2k/qwen2-0.5b-instruction-residuals"
	instruction_residual_filename = "qwen2-0.5b-instruction-residuals.pt"

	class Args:
		def __init__(self):
			self.base_model = base_model
			self.dataset = dataset_path
			self.text_field = text_field
			self.response_field = response_field
			self.prompt_template = "{text}"
			self.response_separator = response_separator
			self.peft_output_dir = peft_output_dir
			self.train_split = "train"
			self.eval_split = None
			self.max_length = max_length
			self.per_device_train_batch_size = per_device_train_batch_size
			self.per_device_eval_batch_size = 1
			self.gradient_accumulation_steps = gradient_accumulation_steps
			self.learning_rate = learning_rate
			self.weight_decay = weight_decay
			self.num_train_epochs = num_train_epochs
			self.warmup_steps = warmup_steps
			self.logging_steps = 10
			self.save_steps = 500
			self.eval_steps = 500
			self.evaluation_strategy = "no"
			self.lora_r = lora_r
			self.lora_alpha = lora_alpha
			self.lora_dropout = 0.05
			self.lora_bias = "none"
			self.lora_target_modules = None
			self.device_map = "auto"
			self.dtype = "auto"
			self.fp16 = fp16
			self.bf16 = bf16
			self.gradient_checkpointing = gradient_checkpointing
			self.seed = 42
			self.max_train_samples = 3000
			self.max_eval_samples = None
			self.num_proc = None
			self.dataset_config = None
			self.instruction_residual_path = str(instruction_residual_path) if instruction_residual_path else None
			self.instruction_residual_repo = instruction_residual_repo
			self.instruction_residual_filename = instruction_residual_filename
			self.hf_token = settings.HF_TOKEN
			self.preprocess = DEPARTMENT_DATASETS["engineering"].get("preprocess")