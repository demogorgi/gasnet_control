echo "usage: ./pltit.sh variable_from_solution_file test_case"
echo "example: ./pltit.sh \"p\[A,B_aux\" rtest2"
echo "grep $1 instances/$2/output/*.sol | grep -Eo ' (.+)$' > dat && gnuplot -p < plt"
echo "grep $1 instances/$2/output/*.sol | grep -Eo ' (.+)$' > dat && gnuplot -p < plt"
grep $1 instances/$2/output/*.sol | grep -Eo ' (.+)$' > dat && gnuplot -p < plt
