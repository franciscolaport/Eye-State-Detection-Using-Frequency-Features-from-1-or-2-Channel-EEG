%% FUNCION PARA LEER GRABACIONES DE EEG DE AMBAS PLACAS:

function [open, closed] = readFiles(sujeto, device, filt)
    
    % Variables generales de la funcion.
    
    task_duration = 58;
    num_tasks = 10;
    experiment_duration = task_duration*num_tasks;
    channels = [2,3];
    
    if strcmp(device, 'PP')
        ini = 0;
        fs = 200;
        path = strcat(strcat('Data/', sujeto),'/PP/');
        extension = '.csv';
    end
    
    % Cargamos el archivo.
    file = strcat(strcat(path,sujeto),extension);
    signal = load(file);
    aux = signal;

    
    % Seleccionamos los datos utiles de la grabacion.
    inicio = ini*fs+1;
    fin = inicio+experiment_duration*fs-1;
    signal = signal(:,channels);
    signal = bsxfun(@plus, signal, - mean(signal,1));
    if filt == 1
        signal = filter(bandpass(fs), signal);
    end
    

    % Empiezan con ojos abiertos.
    state = aux(1,4);
    ini = 1;
    fin = 1;
    cont_op = 1;
    cont_cl = 1;

    for i = 1:num_tasks
        labels = find (aux(:,4) == abs(state-1));
        ini = fin;
        fin = min(labels(labels > ini));
        
        ss = signal(ini:ini+task_duration*fs-1,:);
        if state == 1
            open(cont_op,:,:) = ss;
            cont_op = cont_op+1;
        else
            closed(cont_cl,:,:) = ss;
            cont_cl = cont_cl+1;
        end

        state = abs(state-1);
    end       
    
end