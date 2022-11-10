
package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

@Data
public class AnnotationDeclaration extends Declaration {
    private Name name;
    public ParametersExpression parameters;
    public AnnotationDeclaration(Node prarent) {
        super(prarent);
    }

}
