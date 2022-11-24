package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

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
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (a.equals(AT) && b instanceof Name) {
                AnnotationDeclaration declare = new AnnotationDeclaration(node.getPrarent());
                declare.setName((Name) b);
                int index = node.getChildrens().indexOf(b);
                if (index < node.getChildrens().size()) {
                    Object n = node.getChildrens().get(index + 1);
                    if (n instanceof ParametersExpression) {
                        declare.setParameters((ParametersExpression) n);
                        node.getChildrens().remove(n);
                    }
                }
                node.getChildrens().remove(b);
                node.replace(a, declare);
            }
        });
    }

}
