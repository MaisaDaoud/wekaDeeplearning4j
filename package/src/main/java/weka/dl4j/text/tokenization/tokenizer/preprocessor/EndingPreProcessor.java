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
 *    EndingPreProcessor.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.dl4j.text.tokenization.tokenizer.preprocessor;

import weka.core.Option;
import weka.core.OptionHandler;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * A serializable version of DeepLearning4j's EndingPreProcessor.
 *
 * @author Felipe Bravo-Marquez
 */
public class EndingPreProcessor
    extends org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor
    implements Serializable, OptionHandler {

  /** For Serialization */
  private static final long serialVersionUID = -7863874149371478868L;

  /**
   * Returns a string describing this object.
   *
   * @return a description of the object suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Gets rid of endings: ed,ing, ly, s, ..\n";
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
}
