
package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.AT;

@Data
public class AnnotationDeclaration extends Declaration {
    private Name name;
    public ParametersExpression parameters;
    public AnnotationDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof AnnotationDeclaration) return;
        List.copyOf(node.getChildrens()).stream().reduce((a, b) -> {
            if (a.equals(AT) && b instanceof Name) {
                AnnotationDeclaration annotationDeclare = new AnnotationDeclaration(node.getPrarent());
                annotationDeclare.setName((Name) b);
                int index = node.getChildrens().indexOf(b);
                if (index < node.getChildrens().size()) {
                    Object n = node.getChildrens().get(index + 1);
                    if (n instanceof ParametersExpression) {
                        annotationDeclare.setParameters((ParametersExpression) n);
                        node.remove(n);
                    }
                }
                node.remove(b);
                node.replace(a, annotationDeclare);
            }
            return b;
        });
    }

}
