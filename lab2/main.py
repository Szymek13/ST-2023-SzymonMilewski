import gradio as gr

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_line (file_path,line_number):
    with open(file_path, "r") as file:
        line = file.readlines()[line_number - 1]
    return line

def get_file_info(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    number_lines = len(lines)
    text_lines = read_text_file(file_path)
    info = f"Plik zawiera {number_lines} linii."
    classes = {}
    for line in text_lines:
        class_name = line.split()[0]
        if class_name not in classes:
            classes[class_name] = 0
        classes[class_name] += 1
    return classes, info, number_lines


def display_file(file_path, num_lines):
    classes, info , number_lines= get_file_info(file_path)
    response = "Liczba klas decyzyjnych: {}\n".format(len(classes))
    num_lines = int(num_lines)
    line = "Wartości podanej linii: {}\n".format(read_line(file_path,num_lines))
    return f"{info}\n{response}\n{line}"

file_name_input = gr.inputs.Textbox(label="Nazwa pliku:")
num_lines_input = gr.inputs.Number(label="Liczba linii:")
output_text = gr.outputs.Textbox(label="Content:")

bot = gr.Interface(fn=display_file, inputs=[file_name_input, num_lines_input], outputs=output_text, title="ChatBot", description="Wpisz nazwę pliku oraz liczbę linii do wyświetlenia.")
bot.launch()