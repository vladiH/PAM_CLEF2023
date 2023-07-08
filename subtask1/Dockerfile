FROM python:3.10

RUN pip3 install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip3 install pandas emoji transformers docker

COPY random-baseline-subtask-1.py /random-baseline-subtask-1.py 
COPY weights.pth /weights.pth 
COPY tokenizer /tokenizer

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.10/site-packages/tira.py'

ENTRYPOINT [ "python", "/random-baseline-subtask-1.py" ]