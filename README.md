Скачать по ссылке https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF любую подходящую модель, но лучше google_gemma-3-12b-it-IQ3_XXS.gguf, а также во вкладке файлы И версии скачать mmproj-google_gemma-3-12b-it-f16.gguf и закинуть их в папку Neuro LLM

Чтобы запустить llama-server, нужно скачать и распаковать его в новую папку в проекте, например, "E:\codes\Neuro 2.0\llama.cpp\llama-b12242-bin-win-vulkan-x64", а данном случае Neuro 2.0 - это папка проект, а llama-vulcan - это сборка под мой компьютер, чтобы найти нужную сборку, например с CUDA, нужно скачать его по ссылке https://github.com/ggml-org/llama.cpp/releases

Далее открыть консоль в VS И написать строки в зависимости какие у вас пути к файлам, в моем случае: & "E:\codes\Neuro 2.0\llama.cpp\llama-b12242-bin-win-vulkan-x64\llama-server.exe" -m "E:\codes\Neuro 2.0\Neuro LLM\google_gemma-3-12b-it-IQ3_XXS.gguf" --mmproj "E:\codes\Neuro 2.0\Neuro LLM\mmproj-google_gemma-3-12b-it-f16.gguf" --port 8080 -c 4096 -ngl 40

Далее просто запустить файл main, если не возникло ошибок