import tensorflow as tf
import os

# Путь к директории с чекпоинтом
checkpoint_dir = "models/1558M/"
checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")

# Создание индексного файла для чтения чекпоинта
reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)

# Извлечение всех переменных и их значений
variables = reader.get_variable_to_shape_map()

for variable_name in variables:
    print("Variable name:", variable_name)
    print("Shape:", variables[variable_name])

    variable_value = reader.get_tensor(variable_name)
    print("Value:", variable_value)