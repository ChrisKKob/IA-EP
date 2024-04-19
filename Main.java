import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main {

    //metodo para printar um vetor de caracteres
    public static void printVetor(char[] rotulo){
        for(int i = 0; i < rotulo.length; i++){
            System.out.println(rotulo[i]);
        }
    }

    //metodo para printar uma matriz
    public static void printMatriz(float[][] entrada){
        for(int i = 0; i < entrada.length; i++){
            System.out.println("valor da linha "+ (i+1) +" : ");
            for(int j = 0; j < entrada[i].length; j++){
                System.out.print(entrada[i][j]+" ");
            }
            System.out.println();
        }
    }

    //metodo que captura a entrada de dados e coloca na matriz entrada
    public static void sensor(float[][] entrada) throws IOException{
        BufferedReader leitor = new BufferedReader(new FileReader("X.txt"));
        String linha = "";
        String[] aux = new String[120];
        int linhaNumero = 0;

        while (true) {
            linha = leitor.readLine();

            if(linha != null){
                //System.out.println(linhaNumero);
            }else break;
        
            aux = linha.split(", ");

            for(int i = 0; i < 120; i++){
                entrada[linhaNumero][i] = Float.parseFloat(aux[i]);
            }

            linhaNumero++;
        }
        leitor.close();
    }

    //metodo usado para a leitura do arquivo do rotulo
    public static void captarRotulo(char[] rotulo)throws IOException{
        BufferedReader leitor = new BufferedReader(new FileReader("Y_letra.txt"));
        int indice = 0;
        String linha = "";

        while(true){
            linha = leitor.readLine();

            if (linha != null) {
                System.out.println(linha);
            }else break;

            rotulo[indice] = linha.charAt(0);
            indice++;
        }
        leitor.close();
    }

    public static void main(String[] args) {
        //matriz de entrada dos dados
        float[][] entrada = new float[1326][120];

        //matriz do rotulo da amostra
        char[] rotulo = new char[1326];

        //metodo que captura a entrada de dados e coloca na matriz entrada
        try {
            sensor(entrada);
            captarRotulo(rotulo);
        } catch (Exception e) {
            System.out.println("erro ao ler arquivo");
        }

        //* métodos para verificar se os arquivos foram lidos corretamente
        //printMatriz(entrada);
        //printVetor(rotulo);

    }
}
