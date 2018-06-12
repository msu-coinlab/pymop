
% path to python3
python_path = '/Users/julesy/anaconda3/bin/python';

% path to the framework implementation
pymop_path = '/Users/julesy/workspace/problems-python/pymop/cmd.py';

n_var = 2;
X = rand(100,n_var);
[F,G] = evaluate('TNK', [], X);
test2 = info('TNK', []);
P = pareto_front('DTLZ4', [10,3]);
display(P);



function val = get_problem_str(problem, params)
    str_params = strjoin(arrayfun(@(x) num2str(x),params,'UniformOutput',false),',');
    if size(params) > 0
        problem = sprintf('%s(%s)', problem, str_params);
    end
    val = sprintf('''%s''', problem);
end



function val = info(problem, params)
    problem = get_problem_str(problem, params);
    command = sprintf('%s %s %s info _ _', evalin('base', 'python_path'), evalin('base', 'pymop_path'), problem);
    [status, stdout] = system(command);
    if status ~= 0
        display(stdout);
        error('Error while receiving information!')
    end
    val = jsondecode(stdout);
end



function P = pareto_front(problem, params)
    problem = get_problem_str(problem, params);
    fname = tempname;
    outfile = sprintf('%s.out', fname);
    command = sprintf('%s %s %s front _ %s', evalin('base', 'python_path'), evalin('base', 'pymop_path'), problem, outfile);
    [status, stdout] = system(command);
    if status ~= 0
        display(stdout);
        error('Error while receiving pareto front!')
    end
    P = dlmread(outfile,' ');
end



% evaluate a test function given a the problem and parameter
function [F,G] = evaluate(problem, params, X)

    fname = tempname;
    infile = sprintf('%s.in', fname);
    outfile = sprintf('%s.out', fname);
    dlmwrite(infile,X, 'delimiter', ' ', 'precision', 16);

    problem = get_problem_str(problem, params);
    command = sprintf('%s %s ''%s'' evaluate %s %s', evalin('base', 'python_path'), evalin('base', 'pymop_path'), problem, infile, outfile);
    status = system(command);

    % check if the execution was sucessfully
    if status ~= 0
        error('Error while evaluating!')
    end

    % read the output data
    M = dlmread(outfile,' ');

    s = info(problem, params);
    if s.n_constr == 0
        F = M;
    else
        F = M(:,s.n_var);
        G = M(:,s.n_var+1:end);
    end

end
