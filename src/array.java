import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class array {

    public static void main(String[] args) {
        String fileName= "C:\\Users\\Hp\\OneDrive\\Desktop\\Programs\\Minor Project\\MINOR 2\\output.csv";
        File file= new File(fileName);

        // this gives you a 2-dimensional array of strings
        List<List<String>> lines = new ArrayList<>();
        Scanner inputStream;

        try{
            inputStream = new Scanner(file);

            while(inputStream.hasNext()){
                String line= inputStream.next();
                String[] values = line.split(",");
                // this adds the currently parsed line to the 2-dimensional string array
                lines.add(Arrays.asList(values));
            }

            inputStream.close();
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // the following code lets you iterate through the 2-dimensional array
        int lineNo = 1;
        for(List<String> line: lines) {
            
            int columnNo = 1;
            
            for (String value: line) {
                
                 System.out.println("Line " + lineNo + " Column " + columnNo + ": " + value );
                 if(value.equals("1") )
                 System.out.println("Sussessful");
                 else
                 System.out.println(" Not Sussessful");
                columnNo++;
            }
            lineNo++;
        }  
    
    
    }

}