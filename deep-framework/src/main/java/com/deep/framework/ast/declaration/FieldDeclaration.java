package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.stream.Collectors;

import static com.deep.framework.ast.lexer.TokenType.ASSIGN;

@Data
public class FieldDeclaration extends Declaration {
    private List<TokenType> modifiers;
    private Type type;
    private Name name;
    private Expression initializer;

    public FieldDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof ClassOrInterfaceDeclaration)) return;
        if (node instanceof FieldDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.isName() && n.isName() && !a.endsTypeof(ParametersExpression.class)) {
                    FieldDeclaration declare = new FieldDeclaration(node.getPrarent());
                    List<Node> modifiers = a.getChildrens().stream().filter(e -> Field_Modifiers.contains(e.getTokenType())).toList();
                    a.getChildrens().removeAll(modifiers);
                    a.getChildrens().removeAll(List.of(m, n));
                    declare.setModifiers(modifiers.stream().map(Node::getTokenType).collect(Collectors.toList()));

                    Expression expression = new Expression(declare);
                    expression.setChildrens(a.getChildrens());
                    Node first = expression.getChildrens().first();
                    if (first.equals(ASSIGN)) expression.getChildrens().remove(first);

                    declare.setModifiers(modifiers.stream().map(Token::getTokenType).collect(Collectors.toList()));
                    declare.setType(Type.getType(m));
                    declare.setName((Name) n);
                    declare.setInitializer(expression);
                    declare.setChildrens(Stream.of(declare.getType(), declare.getName(), expression));
                    node.replace(a, declare);
                    c.clear();
                }
            });
        });
    }
}
