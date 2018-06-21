

classdef Problem
   
   properties (Constant)
       % path to python3
      python_path = '/Users/julesy/anaconda3/bin/python';

      % path to the framework implementation
      pymop_path = '/Users/julesy/workspace/problems-python/pymop/cmd.py';
      
   end
    
   properties

      n_var;
      n_obj;
      n_constr;
      xl;
      xu;
      prefix;
      
   end
   methods
       
      function obj = Problem(name, varargin)
        if nargin == 0
            error('Please provide the name of the problem');
        end
        
        params = '[';
        
        n_args = size(varargin, 2);
        
        if rem(n_args,2) == 0
            for i = 1:n_args/2
                key = varargin(2*(i-1) + 1);
                val = varargin(2*(i-1) + 2);
                
                if i > 1
                    params = sprintf('%s,', params);
                end
                params = sprintf('%s "%s", %s', params, key{1}, num2str(val{1}));
       
            end
            params = sprintf('%s]', params);
            
        else
            error('Please provide alternating the properties of the problem.');
        end
        
        obj.prefix = sprintf('%s %s %s "%s"', Problem.python_path, Problem.pymop_path, name, params);
        
        [status, stdout] = system(sprintf('%s info', obj.prefix));
        if status ~= 0
            display(stdout);
            error('Error while initializing the problem!')
        end
        val = jsondecode(stdout);
        
        obj.n_var = val.n_var;
        obj.n_obj = val.n_obj;
        obj.n_constr = val.n_constr;
        obj.xl = val.xl';
        obj.xu = val.xu';

      end
      
      function [F, G] = evaluate(obj, X)
          
        fname = tempname;
        outfile = sprintf('%s.out', fname);  
        dlmwrite(outfile,X, 'delimiter', ' ', 'precision', 16);
          
        [status, stdout] = system(sprintf('%s evaluate < %s', obj.prefix, outfile));
        if status ~= 0
            display(stdout);
            error('Error while initializing the problem!')
        end
        
        s = '%f';
        for i = 1:obj.n_obj + obj.n_constr - 1
            s = sprintf('%s %s', s, '%f');  
        end
        
        M = textscan(stdout, s,'Delimiter',' ');
        
        F = zeros(size(X, 1), obj.n_obj);
        G = zeros(size(X, 1), obj.n_constr);
        
        for i = 1:obj.n_obj
            F(:,i) = M{i};
        end
        
        for i = 1:obj.n_constr
            G(:,i) = M{obj.n_obj + 1};
        end
        
        
      end
      

   end
end

