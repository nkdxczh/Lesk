/**
 * Implement the Lesk algorithm for Word Sense Disambiguation (WSD)
 */
import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import javafx.util.Pair;
import java.net.*;

import edu.mit.jwi.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.item.*; 

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;

public class Lesk {

    /** 
     * Each entry is a sentence where there is at least a word to be disambiguate.
     * E.g.,
     * 		testCorpus.get(0) is Sentence object representing
     * 			"It is a full scale, small, but efficient house that can become a year' round retreat complete in every detail."
     **/
    private ArrayList<Sentence> testCorpus = new ArrayList<Sentence>();

    /** Each entry is a list of locations (integers) where a word needs to be disambiguate.
     * The index here is in accordance to testCorpus.
     * E.g.,
     * 		ambiguousLocations.get(0) is a list [13]
     * 		ambiguousLocations.get(1) is a list [10, 28]
     **/
    private ArrayList<ArrayList<Integer> > ambiguousLocations = new ArrayList<ArrayList<Integer> >();

    /**
     * Each entry is a list of pairs, where each pair is the lemma and POS tag of an ambiguous word.
     * E.g.,
     * 		ambiguousWords.get(0) is [(become, VERB)]
     * 		ambiguousWords.get(1) is [(take, VERB), (apply, VERB)]
     */
    private ArrayList<ArrayList<Pair<String, String> > > ambiguousWords = new ArrayList<ArrayList<Pair<String, String> > > (); 

    /**
     * Each entry is a list of maps, each of which maps from a sense key to similarity(context, signature)
     * E.g.,
     * 		predictions.get(1) = [{take%2:30:01:: -> 0.9, take%2:38:09:: -> 0.1}, {apply%2:40:00:: -> 0.1}]
     */
    private ArrayList<ArrayList<HashMap<String, Double> > > predictions = new ArrayList<ArrayList<HashMap<String, Double> > >();

    /**
     * Each entry is a list of ground truth senses for the ambiguous locations.
     * Each String object can contain multiple synset ids, separated by comma.
     * E.g.,
     * 		groundTruths.get(0) is a list of strings ["become%2:30:00::,become%2:42:01::"]
     * 		groundTruths.get(1) is a list of strings ["take%2:30:01::,take%2:38:09::,take%2:38:10::,take%2:38:11::,take%2:42:10::", "apply%2:40:00::"]
     */
    private ArrayList<ArrayList<String> > groundTruths = new ArrayList<ArrayList<String> >();

    /* This section contains the NLP tools */

    private Set<String> POSSet = new HashSet<String>(Arrays.asList("ADJECTIVE", "ADVERB", "NOUN", "VERB"));

    private IDictionary wordnetdict;

    private StanfordCoreNLP pipeline;

    private Set<String> stopwords;

    public Lesk() {

        stopwords = new HashSet<>();

        try{
            FileReader fileReader = new FileReader("data/stopwords.txt");

            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line = null;

            while((line = bufferedReader.readLine()) != null) {
                stopwords.add(line);
            }

            bufferedReader.close();
        } catch (Exception e) {
        }

        try{
            wordnetdict = new Dictionary(new URL("file", null, "data/dict"));
            wordnetdict.open();
        }catch(Exception e){}

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        pipeline = new StanfordCoreNLP(props);

    }

    /**
     * Convert a pos tag in the input file to a POS tag that WordNet can recognize (JWI needs this).
     * We only handle adjectives, adverbs, nouns and verbs.
     * @param pos: a POS tag from an input file.
     * @return JWI POS tag.
     */
    private String toJwiPOS(String pos) {
        if (pos.equals("ADJ")) {
            return "ADJECTIVE";
        } else if (pos.equals("ADV")) {
            return "ADVERB";
        } else if (pos.equals("NOUN") || pos.equals("VERB")) {
            return pos;
        } else {
            return null;
        }
    }

    /**
     * This function fills up testCorpus, ambiguousLocations and groundTruths lists
     * @param filename
     */
    public void readTestData(String filename) throws Exception {
        FileReader fileReader = new FileReader(filename);

        BufferedReader bufferedReader = new BufferedReader(fileReader);

        String line = null;

        while((line = bufferedReader.readLine()) != null) {
            String text = line;

            Annotation document = new Annotation(text);
            pipeline.annotate(document);
            List<CoreMap> sentences = document.get(SentencesAnnotation.class);
            for(CoreMap sentence: sentences) {
                Sentence s = new Sentence();
                for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                    String word = token.get(LemmaAnnotation.class).toLowerCase();
                    String pos = token.get(PartOfSpeechAnnotation.class);

                    Word w = new Word(word);
                    w.setPosTag(pos);
                    s.addWord(w);
                }
                testCorpus.add(s);
            }


            int word_num = Integer.parseInt(bufferedReader.readLine());

            ArrayList<Integer> locations = new ArrayList<>();
            ArrayList<Pair<String, String> > words = new ArrayList<>();
            ArrayList<String> truths = new ArrayList<>();

            for(int i = 0; i < word_num; ++i){
                line = bufferedReader.readLine();
                String[] ele = line.split(" ");

                locations.add(Integer.parseInt(ele[0].substring(1)));
                words.add(new Pair<String, String>(ele[1], ele[2]));
                truths.add(ele[3]);
            }

            ambiguousLocations.add(locations);
            ambiguousWords.add(words);
            groundTruths.add(truths);
        }   

        bufferedReader.close(); 
    }

    private POS getPartOfSpeech(String postag) {

        if(postag.equals("NOUN")) {
            return POS.NOUN;
        }
        if (postag.equals("VERB")) {
            return POS.VERB;
        }
        if (postag.equals("ADJECTIVE")) {
            return POS.ADJECTIVE;                   
        }
        if (postag.equals("ADVERB")) {
            return POS.ADVERB;
        }
        return null;
    }
    /**
     * Create signatures of the senses of a pos-tagged word.
     * 
     * 1. use lemma and pos to look up IIndexWord using Dictionary.getIndexWord()
     * 2. use IIndexWord.getWordIDs() to find a list of word ids pertaining to this (lemma, pos) combination.
     * 3. Each word id identifies a sense/synset in WordNet: use Dictionary's getWord() to find IWord
     * 4. Use the getSynset() api of IWord to find ISynset
     *    Use the getSenseKey() api of IWord to find ISenseKey (such as charge%1:04:00::)
     * 5. Use the getGloss() api of the ISynset interface to get the gloss String
     * 6. Use the Dictionary.getSenseEntry(ISenseKey).getTagCount() to find the frequencies of the synset.d
     * 
     * @param args
     * lemma: word form to be disambiguated
     * pos_name: POS tag of the wordform, must be in {ADJECTIVE, ADVERB, NOUN, VERB}.
     * 
     */
    private Map<String, Pair<String, Integer> > getSignatures(String lemma, String pos_name) {
        Map<String, Pair<String, Integer> > map = new HashMap<String, Pair<String, Integer> >();
        if(pos_name == null){
            System.out.println(lemma + "," + pos_name);
            return map;
        }
        IIndexWord idxWord = wordnetdict.getIndexWord(lemma, getPartOfSpeech(pos_name));
        if(idxWord == null){
            System.out.println(lemma + "," + pos_name);
            return map;
        }
        List<IWordID> wordID = idxWord.getWordIDs();
        for(IWordID id : wordID){
            IWord word = wordnetdict.getWord(id);
            ISynset synset = word.getSynset();
            ISenseKey sensekey = word.getSenseKey();
            String gloss = synset.getGloss();
            int count = wordnetdict.getSenseEntry(sensekey).getTagCount();

            Pair<String, Integer> pair = new Pair(gloss, count);
            map.put(sensekey + "", pair);
        }
        return map;
    }

    /**
     * Create a bag-of-words representation of a document (a sentence/phrase/paragraph/etc.)
     * @param str: input string
     * @return a list of strings (words, punctuation, etc.)
     */
    private ArrayList<String> str2bow(String str) {
        ArrayList<String> res = new ArrayList<>();
        String[] words = str.split(" ");
        for(String word : words){
            res.add(word);
        }
        return res;
    }

    /**
     * compute similarity between two bags-of-words.
     * @param bag1 first bag of words
     * @param bag2 second bag of words
     * @param sim_opt COSINE or JACCARD similarity
     * @return similarity score
     */
    private double similarity(ArrayList<String> bag1, ArrayList<String> bag2, String sim_opt) {
        if(sim_opt.equals("COSINE")){
            Set<String> overlap = new HashSet<String>(bag1);
            Set<String> set2 = new HashSet<String>(bag2);

            overlap.retainAll(set2);

            return (double)overlap.size() / (Math.sqrt(bag1.size()) * Math.sqrt(bag2.size()));
        }
        if(sim_opt.equals("JACCARD")){
            Set<String> overlap = new HashSet<String>(bag1);
            Set<String> merge = new HashSet<String>(bag1);
            Set<String> set2 = new HashSet<String>(bag2);

            overlap.retainAll(set2);
            merge.addAll(set2);

            return (double)overlap.size() / merge.size();
        }
        return 0;
    }

    /**
     * This is the WSD function that prediction what senses are more likely.
     * @param context_option: one of {ALL_WORDS, ALL_WORDS_R, WINDOW, POS}
     * @param window_size: an odd positive integer > 1
     * @param sim_option: one of {COSINE, JACCARD}
     */
    public void predict(String context_option, int window_size, String sim_option) {
        predictions = new ArrayList<>();

        for(int i = 0; i < ambiguousWords.size(); ++i){
            ArrayList<HashMap<String, Double> > list = new ArrayList<>();
            Sentence sentence = testCorpus.get(i);

            int count = 0;
            for(Pair<String, String> target : ambiguousWords.get(i)){
                Map<String, Pair<String, Integer> > map = getSignatures(target.getKey(), toJwiPOS(target.getValue()));

                ArrayList<String> context = new ArrayList<>();
                if(context_option.equals("ALL_WORDS")){
                    for(int loc = 0; loc < sentence.length(); ++loc){
                        context.add(sentence.getWordAt(loc).getLemme());
                    }
                }
                else if(context_option.equals("ALL_WORDS_R")){
                    for(int loc = 0; loc < sentence.length(); ++loc){
                        if(stopwords.contains(sentence.getWordAt(loc).getLemme()))continue;
                        context.add(sentence.getWordAt(loc).getLemme());
                    }
                }
                else if(context_option.equals("WINDOW")){
                    int center = ambiguousLocations.get(i).get(count);
                    int begin = center - (int)(window_size / 2);
                    int end = center + (int)(window_size / 2);

                    for(int loc = Math.max(0, begin); loc < Math.min(end, sentence.length()); ++loc){
                        context.add(sentence.getWordAt(loc).getLemme());
                    }
                }
                else if(context_option.equals("POS")){
                    for(int loc = 0; loc < sentence.length(); ++loc){
                        if(!POSSet.contains(sentence.getWordAt(loc).getPosTag()))continue;
                        context.add(sentence.getWordAt(loc).getLemme());
                    }
                }
                count++;

                HashMap<String, Double> resmap = new HashMap<>();
                for(String key : map.keySet()){
                    Pair<String, Integer> pair = map.get(key);
                    String gloss = pair.getKey();
                    ArrayList<String> bag= str2bow(gloss);
                    double pro = similarity(context, bag, sim_option);

                    resmap.put(key, pro);
                }
                list.add(resmap);
            }
            predictions.add(list);
        }
    }


    /**
     * Multiple senses are concatenated using comma ",". Separate them out.
     * @param senses
     * @return
     */
    private ArrayList<String> parseSenseKeys(String senseStr) {
        ArrayList<String> senses = new ArrayList<String>();
        String[] items = senseStr.split(",");
        for (String item : items) {
            senses.add(item);
        }
        return senses;
    }

    /*private Set<String> getTopK(HashMap<String, Double> predictions, int K){
      double values[predictions.size()];
      }*/

    /**
     * Precision/Recall/F1-score at top K positions
     * @param groundTruths: a list of sense id strings, such as [become%2:30:00::, become%2:42:01::]
     * @param predictions: a map from sense id strings to the predicted similarity
     * @param K
     * @return a list of [top K precision, top K recall, top K F1]
     */
    private ArrayList<Double> evaluate(ArrayList<String> groundTruths, HashMap<String, Double> predictions, int K) {
        ArrayList<Double> res = new ArrayList<>();
        Set<String> truths = new HashSet<String>(groundTruths);

        Map<String, Double> top =
            predictions.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).limit(K).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        Set<String> overlap = top.keySet();
        overlap.retainAll(truths);

        res.add((double)overlap.size()/K);
        res.add((double)overlap.size()/predictions.size());
        res.add(2*res.get(0)*res.get(1) / (res.get(0) + res.get(1)));

        return res;
    }

    /**
     * Test the prediction performance on all test sentences
     * @param K Top-K precision/recall/f1
     */
    public ArrayList<Double> evaluate(int K) {
        ArrayList<Double> res = new ArrayList<>();
        for(int i = 0; i < groundTruths.size(); ++i){
            for(int j = 0; j < groundTruths.get(i).size(); ++j){
                ArrayList<Double> tem = evaluate(parseSenseKeys(groundTruths.get(i).get(j)), predictions.get(i).get(j), K);
                res.addAll(tem);
            }
        }
        return res;
    }

    /**
     * @param args[0] file name of a test corpus
     */
    public static void main(String[] args) {
        Lesk model = new Lesk();
        try {
            model.readTestData(args[0]);
        } catch (Exception e) {
            System.out.println(args[0]);
            e.printStackTrace();
        }
        //String context_opt = "ALL_WORDS";
        String context_opt = "WINDOWS";
        int window_size = 3;
        //String sim_opt = "JACCARD";
        String sim_opt = "CONSINE";

        model.predict(context_opt, window_size, sim_opt);

        ArrayList<Double> res = model.evaluate(1);
        System.out.print(args[0]);
        System.out.print("\t");
        System.out.print(res.get(0));
        System.out.print("\t");
        System.out.print(res.get(1));
        System.out.print("\t");
        System.out.println(res.get(2));
    }
}
