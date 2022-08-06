import java.util.*;
import java.util.ArrayList;
import java.util.Random;

public class nlpPipeline {
    
    public static String findSentiment(String string) {
        ArrayList<String> list = new ArrayList<>();
        list.add("positive");
        list.add("neutral");
        list.add("negative");
        nlpPipeline obj=new nlpPipeline();
        return obj.getRandomElement(list);}

        public String getRandomElement(List<String> list){
            Random rand = new Random();
            return list.get(rand.nextInt(list.size()));
        }
    
    

    public static void init() {
    }

}
