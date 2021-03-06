/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    TweetNLPTokenizerFactory.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.dl4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.tokenizers.NGramTokenizer;

import java.io.InputStream;
import java.io.Serializable;
import java.util.Enumeration;

/**
 * A DeepLearning4j's TokenizerFactory interface for Weka core tokenizers.
 *
 * @author Felipe Bravo-Marquez
 */
public class NGramTokenizerFactory implements TokenizerFactory, Serializable, OptionHandler {

  /** For Serialization */
  private static final long serialVersionUID = 4694868790645893109L;
  /** the maximum number of N */
  protected int nMax = 3;
  /** the minimum number of N */
  protected int nMin = 1;
  /** Delimiters used in tokenization */
  protected String delimiters = " \r\n\t.,;:'\"()?!";
  /** The TokenPreProcess object */
  private TokenPreProcess tokenPreProcess;
  private NGramTokenizer wekaTokenizer;

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#create(java.lang.String)
   */
  @Override
  public Tokenizer create(String toTokenize) {

    this.wekaTokenizer = new NGramTokenizer();
    this.wekaTokenizer.setNGramMinSize(this.nMin);
    this.wekaTokenizer.setNGramMaxSize(this.nMax);
    this.wekaTokenizer.setDelimiters(this.delimiters);

    WekaTokenizer t = new WekaTokenizer(toTokenize, wekaTokenizer);
    t.setTokenPreProcessor(tokenPreProcess);
    return t;
  }

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#create(java.io.InputStream)
   */
  @Override
  public Tokenizer create(InputStream toTokenize) {
    return null;
  }

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#getTokenPreProcessor()
   */
  @Override
  public TokenPreProcess getTokenPreProcessor() {
    return tokenPreProcess;
  }

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory#setTokenPreProcessor(org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess)
   */
  @Override
  public void setTokenPreProcessor(TokenPreProcess preProcessor) {
    this.tokenPreProcess = preProcessor;
  }

  /**
   * Returns a string describing this object.
   *
   * @return a description of the object suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Splits a string into an n-gram with min and max grams.";
  }

  /* (non-Javadoc)
   * @see weka.core.OptionHandler#listOptions()
   */
  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClass(this.getClass()).elements();
  }

  /* (non-Javadoc)
   * @see weka.core.OptionHandler#getOptions()
   */
  @Override
  public String[] getOptions() {
    return Option.getOptions(this, this.getClass());
  }

  /* (non-Javadoc)
   * @see weka.core.OptionHandler#setOptions(java.lang.String[])
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    Option.setOptions(options, this, this.getClass());
  }

  @OptionMetadata(
    displayName = "NMax",
    description = "NGram max size.",
    commandLineParamName = "NMax",
    commandLineParamSynopsis = "-NMax <int>",
    displayOrder = 0
  )
  public int getNMax() {
    return nMax;
  }

  public void setNMax(int nMax) {
    this.nMax = nMax;
  }

  @OptionMetadata(
    displayName = "NMin",
    description = "NGram min size.",
    commandLineParamName = "NMin",
    commandLineParamSynopsis = "-NMin <int>",
    displayOrder = 1
  )
  public int getNMin() {
    return nMin;
  }

  public void setNMin(int nMin) {
    this.nMin = nMin;
  }

  @OptionMetadata(
    displayName = "delimiters",
    description =
        "Set of delimiter characters to use in tokenizing (\\r, \\n and \\t can be used for carriage-return, line-feed and tab).",
    commandLineParamName = "delimiters",
    commandLineParamSynopsis = "-delimiters <int>",
    displayOrder = 2
  )
  public String getDelimiters() {
    return delimiters;
  }

  public void setDelimiters(String delimiters) {
    this.delimiters = delimiters;
  }
}
