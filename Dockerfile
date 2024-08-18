FROM building_dag

EXPOSE 8502

WORKDIR /root/BuildingDAG24/

ENV PATH=/root/miniconda3/bin:$PATH
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dag", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dag", "streamlit", "run", "d_st_ui.py", "--server.port", "8502"]
