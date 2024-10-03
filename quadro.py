# Importa a biblioteca Tkinter
import tkinter as tk

# Define uma função para ser executada quando o botão for clicado
de botao_clicado():
  # Altera o texto do rótulo para "Botão clicado!"
  label["text"] = "Botão clicado!"

# Cria uma janela Tkinter
janela = tk.Tk(

# Define o título da janela
janela.title("Minha Interface")

# Cria um rótulo com o texto "Olá, mundo!"
label = tk.Label(janela, text="Olá, mundo!")

# Empacota o rótulo na janela
label.pack(

# Cria um botão com o texto "Clique aqui"
botao = tk.Button(janela, text="Clique aqui", command=botao_clicado)

# Empacota o botão na janela
botao.pack()

# Inicia o loop principal da janela Tkinter
janela.loop()