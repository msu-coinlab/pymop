

% check ZDT1
p = Problem("ZDT1");
F = p.evaluate(rand(100,p.n_var));

% check TNK
p = Problem("TNK");
[F,G] = p.evaluate(rand(100,p.n_var));

% check DTLZ1
p = Problem("DTLZ1", "n_var", 10, "n_obj", 5);
[F,G] = p.evaluate(rand(100,p.n_var));

% check Rastrigin
p = Problem("Rastrigin");
[F,G] = p.evaluate(rand(100,p.n_var));

% check OSY
p = Problem("OSY");
[F,G] = p.evaluate(rand(100,p.n_var));

% check Griewank
p = Problem("Griewank");
[F,G] = p.evaluate(rand(100,p.n_var));


% check Kursawe
p = Problem("Kursawe");
[F,G] = p.evaluate(rand(100,p.n_var));
